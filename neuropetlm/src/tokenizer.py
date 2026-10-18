"""A small, deterministic character tokenizer for NeuroPetLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class CharacterTokenizer:
    """Map individual Unicode characters to integer IDs and back.

    Fit the tokenizer on the training text only, then save it alongside model
    checkpoints. Its vocabulary order is deterministic so token IDs remain
    stable when the same training text is used again.
    """

    FORMAT_VERSION = 1
    UNK_TOKEN = "<unk>"

    def __init__(self, vocabulary: list[str]) -> None:
        if not vocabulary or vocabulary[0] != self.UNK_TOKEN:
            raise ValueError("vocabulary must start with the <unk> token")
        if any(not isinstance(token, str) or len(token) != 1 for token in vocabulary[1:]):
            raise ValueError("vocabulary entries after <unk> must be single characters")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary entries must be unique")

        self.vocabulary = list(vocabulary)
        self._char_to_id = {char: index for index, char in enumerate(self.vocabulary)}

    @classmethod
    def fit(cls, text: str) -> "CharacterTokenizer":
        """Build a vocabulary from text, preserving all Unicode characters."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        characters = sorted(set(text))
        # ID 0 is reserved for unknown characters. Sorting makes IDs reproducible.
        return cls([cls.UNK_TOKEN, *characters])

    @property
    def vocab_size(self) -> int:
        """Return the number of entries, including <unk>."""
        return len(self.vocabulary)

    @property
    def unk_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        """Convert text to IDs, mapping unseen characters to ``unk_id``."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [self._char_to_id.get(char, self.unk_id) for char in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Convert IDs to text; unknown ID 0 is rendered as ``<unk>``."""
        decoded: list[str] = []
        for token_id in ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"token ID out of range: {token_id}")
            decoded.append(self.vocabulary[token_id] if token_id else self.UNK_TOKEN)
        return "".join(decoded)

    def save(self, path: str | Path) -> None:
        """Save the vocabulary as readable UTF-8 JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "tokenizer": "character",
            "vocabulary": self.vocabulary,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        """Load a vocabulary saved by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("tokenizer file must contain a JSON object")
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported tokenizer format version")
        if payload.get("tokenizer") != "character":
            raise ValueError("file does not contain a character tokenizer")
        vocabulary = payload.get("vocabulary")
        if not isinstance(vocabulary, list):
            raise ValueError("tokenizer vocabulary must be a list")
        return cls(vocabulary)
