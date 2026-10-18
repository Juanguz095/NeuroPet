"""Trainable token and position embeddings for NeuroPetLM."""

import torch
from torch import nn


class TokenAndPositionEmbedding(nn.Module):
    """Turn token IDs shaped ``(batch, sequence)`` into vectors.

    Token embeddings give each character ID a trainable vector. Position
    embeddings give each location in the context window a trainable vector.
    Their sum is the input representation used by the Transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if context_length <= 0:
            raise ValueError("context_length must be positive")

        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return embeddings shaped ``(batch, sequence, embedding_dim)``."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")

        _, sequence_length = token_ids.shape
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length "
                f"{self.context_length}"
            )

        token_vectors = self.token_embedding(token_ids)
        positions = torch.arange(sequence_length, device=token_ids.device)
        position_vectors = self.position_embedding(positions)
        return token_vectors + position_vectors.unsqueeze(0)
