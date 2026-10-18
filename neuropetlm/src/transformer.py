"""Decoder-only Transformer blocks used by NeuroPetLM."""

import torch
from torch import nn

from neuropetlm.src.attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Two-layer MLP that processes each sequence position independently."""

    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TransformerBlock(nn.Module):
    """One pre-normalized attention + feed-forward residual block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.feed_forward(self.feed_forward_norm(inputs))
