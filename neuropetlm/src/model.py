"""NeuroPetLM: a small decoder-only character language model."""

import torch
from torch import nn

from neuropetlm.config import ModelConfig
from neuropetlm.src.embeddings import TokenAndPositionEmbedding
from neuropetlm.src.transformer import TransformerBlock


class NeuroPetLM(nn.Module):
    """Predict the next character using randomly initialized weights."""

    def __init__(self, vocab_size: int, config: ModelConfig) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if config.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.config = config
        self.vocab_size = vocab_size
        self.embeddings = TokenAndPositionEmbedding(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            context_length=config.context_length,
        )
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    context_length=config.context_length,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.language_head = nn.Linear(config.embedding_dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return next-character logits and optional cross-entropy loss."""
        hidden = self.embeddings(token_ids)
        hidden = self.final_norm(self.blocks(hidden))
        logits = self.language_head(hidden)

        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same shape as token_ids")
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Sample characters one at a time, conditioning on preceding IDs."""
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("generation expects token_ids shaped (1, sequence)")
        if token_ids.shape[1] == 0:
            raise ValueError("generation prompt must contain at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")

        was_training = self.training
        self.eval()
        result = token_ids
        for _ in range(max_new_tokens):
            context = result[:, -self.config.context_length :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                count = min(top_k, self.vocab_size)
                cutoff = torch.topk(next_logits, count, dim=-1).values[:, -1:]
                next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
            probabilities = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            result = torch.cat((result, next_id), dim=1)

        self.train(was_training)
        return result
