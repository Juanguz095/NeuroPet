"""Causal multi-head self-attention for NeuroPetLM."""

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Let each sequence position attend to itself and earlier positions.

    The upper triangle of the attention score matrix is masked, so a token
    cannot use information from later tokens while predicting the next one.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.context_length = context_length

        # A single learned projection produces query, key, and value vectors.
        self.qkv_projection = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Accept and return tensors shaped ``(batch, sequence, embedding_dim)``."""
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, sequence, embedding_dim)")

        batch_size, sequence_length, embedding_dim = inputs.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"expected embedding dimension {self.embedding_dim}, got {embedding_dim}"
            )
        if sequence_length == 0:
            raise ValueError("sequence length must be positive")
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length "
                f"{self.context_length}"
            )

        # Split the final projection into Q, K, V, then split each into heads.
        qkv = self.qkv_projection(inputs)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        # Compare each query with each key. Scaling keeps scores in a useful range.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # True entries are future positions and therefore cannot be attended to.
        future_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=inputs.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(future_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.embedding_dim
        )
        return self.output_dropout(self.output_projection(attended))
