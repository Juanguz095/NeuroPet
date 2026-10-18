"""Small, CPU-friendly defaults for the NeuroPetLM v0.1 prototype."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    context_length: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 10
    learning_rate: float = 3e-4
    train_fraction: float = 0.9
    seed: int = 1337


CHECKPOINT_DIR = "neuropetlm/checkpoints"
DATA_PATH = "neuropetlm/data/conversaciones.txt"
