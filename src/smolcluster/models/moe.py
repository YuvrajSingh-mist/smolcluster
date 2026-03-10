"""
Mixture of Experts (MoE) Transformer implementation (Mixtral-style).

This module contains:
- Mixtral: MoE-based decoder-only transformer with expert routing
- MoeLayer: Sparse mixture of experts layer with top-k routing
- SWiGLUExpertMoE: Expert implementation using SwiGLU activation
"""

from typing import Any, Optional

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
        self, x: torch.Tensor, q: Optional[torch.Tensor] = None, k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for rotary embeddings.

        Args:
            x: Input tensor.
            q: Query tensor (optional).
            k: Key tensor (optional).

        Returns:
            Tensor with rotary embeddings applied.
        """
        return self.apply_rope(x, q, k)


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


class MoeLayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""

    def __init__(
        self,
        embeddings_dims: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        noisy_topk: bool = False,
        use_checkpointing: bool = False,
    ):
        """Initialize MoE layer.

        Args:
            embeddings_dims: Dimension of embeddings.
            num_experts: Number of experts.
            top_k: Number of experts to route to.
            device: Device to place tensors on.
            noisy_topk: Whether to use noisy top-k gating.
            use_checkpointing: Whether checkpointing is used.

        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_topk = noisy_topk
        self.use_checkpointing = use_checkpointing

        self.heads = nn.ModuleList(
            [
                SWiGLUExpertMoE(embeddings_dims=embeddings_dims, device=device)
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Linear(
            in_features=embeddings_dims, out_features=num_experts, device=device, bias=False
        )
        if noisy_topk and not use_checkpointing:
            self.noise = nn.Linear(
                in_features=embeddings_dims, out_features=num_experts, device=device, bias=False
            )
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for MoE layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embeddings_dims).

        Returns:
            Output tensor after expert routing and combination.
        """
        gate_out = self.gate(x)  # [batch_size, seq_len, num_experts]

        if self.noisy_topk and not self.use_checkpointing:
            noise = self.noise(x)
            gaussian_noise = torch.normal(0, 1, size=gate_out.shape, device=self.device)
            noisy_router = F.softplus(noise) * gaussian_noise
            noisy_router += gate_out
        else:
            noisy_router = gate_out

        top_k_values, top_k_indices = torch.topk(noisy_router, k=self.top_k)  # [batch_size, seq_len, top_k]
        probs = F.softmax(top_k_values, dim=-1)  # [batch_size, seq_len, top_k]

        out = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            # Create mask for current expert across all top_k positions
            expert_mask = top_k_indices == expert_idx

            # Sum probabilities for current expert
            expert_weights = (probs * expert_mask).sum(dim=-1)  # [batch_size, seq_len]

            # Get inputs where expert is used
            selected = expert_weights > 0
            if not selected.any():
                continue

            # Process all selected inputs through expert
            expert_out = self.heads[expert_idx](x[selected])

            # Weight and accumulate outputs
            out[selected] += expert_out * expert_weights[selected].unsqueeze(-1)

        return out


class AttentionHead(nn.Module):
    """Single attention head with optional flash attention and RoPE."""

    def __init__(
        self,
        embeddings_dims: int,
        no_of_heads: int,
        device: torch.device,
        attn_dropout: float = 0.1,
        use_flash_attention: bool = True,
    ):
        """Initialize attention head.

        Args:
            embeddings_dims: Dimension of embeddings.
            no_of_heads: Number of attention heads.
            device: Device to place tensors on.
            attn_dropout: Dropout probability for attention.
            use_flash_attention: Whether to use flash attention.
        """
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.no_of_heads = no_of_heads
        self.use_flash_attention = use_flash_attention
        self.dropout_p = attn_dropout
        self.device = device

        if not use_flash_attention:
            self.query = nn.Linear(
                in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False
            )
            self.keys = nn.Linear(
                in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False
            )
            self.values = nn.Linear(
                in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False
            )
            self.rotary = RotaryEmbeddings(
                embeddings_dims=self.head_size, device=device, block_size=2048
            )
        else:
            self.qkv_proj = nn.Linear(embeddings_dims, 3 * embeddings_dims, bias=False, device=device)
            self.rotary = RotaryEmbeddings(
                embeddings_dims=embeddings_dims, device=device, block_size=2048
            )

        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for attention head.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embeddings_dims).

        Returns:
            Output tensor after attention.
        """
        batch_size, block_size, embd_dims = x.shape

        if not self.use_flash_attention:
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
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch_size, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            k = k.view(batch_size, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            v = v.view(batch_size, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            q, k = self.rotary(x, q, k)

            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, is_causal=True)

            # Properly merge heads
            out = out.transpose(1, 2).contiguous().view(batch_size, block_size, -1)
            return out
        
        # Should not reach here, but return empty tensor for type safety
        return torch.empty(0)


class MHA(nn.Module):
    """Multi-head attention."""

    def __init__(
        self,
        embeddings_dims: int,
        no_of_heads: int,
        device: torch.device,
        attn_dropout: float = 0.1,
        use_flash_attention: bool = True,
    ):
        """Initialize multi-head attention.

        Args:
            embeddings_dims: Dimension of embeddings.
            no_of_heads: Number of attention heads.
            device: Device to place tensors on.
            attn_dropout: Dropout probability for attention.
            use_flash_attention: Whether to use flash attention.
        """
        super().__init__()
        self.no_of_heads = no_of_heads
        self.use_flash_attention = use_flash_attention
        
        if use_flash_attention:
            # Flash attention handles all heads internally, so we only need one AttentionHead
            self.heads = nn.ModuleList(
                [
                    AttentionHead(
                        embeddings_dims=embeddings_dims,
                        no_of_heads=no_of_heads,
                        device=device,
                        attn_dropout=attn_dropout,
                        use_flash_attention=True,
                    )
                ]
            )
        else:
            # Non-flash attention uses separate heads
            self.heads = nn.ModuleList(
                [
                    AttentionHead(
                        embeddings_dims=embeddings_dims,
                        no_of_heads=no_of_heads,
                        device=device,
                        attn_dropout=attn_dropout,
                        use_flash_attention=False,
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
    """Transformer decoder block with MoE."""

    def __init__(
        self,
        embeddings_dims: int,
        no_of_heads: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        noisy_topk: bool = False,
        use_checkpointing: bool = False,
        use_flash_attention: bool = True,
    ):
        """Initialize transformer decoder block.

        Args:
            embeddings_dims: Dimension of embeddings.
            no_of_heads: Number of attention heads.
            num_experts: Number of experts in MoE layer.
            top_k: Number of experts to route to.
            device: Device to place tensors on.
            attn_dropout: Dropout probability for attention.
            dropout: General dropout probability.
            noisy_topk: Whether to use noisy top-k gating.
            use_checkpointing: Whether checkpointing is used.
            use_flash_attention: Whether to use flash attention.

        """
        super().__init__()
        self.mha = MHA(
            embeddings_dims=embeddings_dims,
            no_of_heads=no_of_heads,
            device=device,
            attn_dropout=attn_dropout,
            use_flash_attention=use_flash_attention,
        )
        self.layer_norm1 = LayerNormalization(embeddings_dims)
        self.layer_norm2 = LayerNormalization(embeddings_dims)
        self.moe_block = MoeLayer(
            embeddings_dims=embeddings_dims,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
            noisy_topk=noisy_topk,
            use_checkpointing=use_checkpointing,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for transformer decoder block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after attention and MoE.
        """
        x = x + self.mha(self.layer_norm1(x))
        x = x + self.moe_block(self.layer_norm2(x))
        return x


class Mixtral(nn.Module):
    """Mixtral: MoE-based decoder-only transformer."""

    def __init__(
        self,
        vocab_size: int,
        embeddings_dims: int = 768,
        no_of_heads: int = 12,
        no_of_decoder_layers: int = 12,
        num_experts: int = 8,
        top_k: int = 2,
        max_seq_len: int = 1024,
        device: torch.device = torch.device("cpu"),
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        noisy_topk: bool = False,
        use_checkpointing: bool = False,
        use_flash_attention: bool = True,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize Mixtral model.

        Args:
            vocab_size: Size of vocabulary.
            embeddings_dims: Dimension of embeddings.
            no_of_heads: Number of attention heads.
            no_of_decoder_layers: Number of decoder layers.
            num_experts: Number of experts in each MoE layer.
            top_k: Number of experts to route to.
            max_seq_len: Maximum sequence length.
            device: Device to place tensors on.
            attn_dropout: Dropout probability for attention.
            dropout: General dropout probability.
            noisy_topk: Whether to use noisy top-k gating.
            use_checkpointing: Whether to use gradient checkpointing.
            use_flash_attention: Whether to use flash attention.
            tokenizer: Tokenizer for loss computation (optional).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings_dims = embeddings_dims
        self.max_seq_len = max_seq_len
        self.use_checkpointing = use_checkpointing
        self.tokenizer = tokenizer

        self.text_embds = TextEmbeddings(
            vocab_size=vocab_size, embeddings_dims=embeddings_dims, device=device
        )
        self.linear_layer = nn.Linear(
            in_features=embeddings_dims, out_features=vocab_size, device=device, bias=False
        )
        self.layer_norm = LayerNormalization(embeddings_dims=embeddings_dims)
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embeddings_dims=embeddings_dims,
                    no_of_heads=no_of_heads,
                    num_experts=num_experts,
                    top_k=top_k,
                    device=device,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    noisy_topk=noisy_topk,
                    use_checkpointing=use_checkpointing,
                    use_flash_attention=use_flash_attention,
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

    def forward(
        self, x: torch.Tensor 
        ) -> torch.Tensor:
        """Forward pass for Mixtral.

        Args:
            x: Input token indices of shape (batch_size, seq_len).
            actual_labels: Target labels for loss computation (optional).
            inference: Whether in inference mode.

        Returns:
            Logits or loss depending on mode.
        """
        x = self.text_embds(x)

    
        for layer in self.decoder_layers:
            x = layer(x)

        x = self.layer_norm(x)

        out = self.linear_layer(x)
        return out
    