"""Train NeuroPetLM from scratch on the bundled character corpus."""

from pathlib import Path

import torch

from neuropetlm.config import (
    CHECKPOINT_DIR,
    DATA_PATH,
    ModelConfig,
    TrainingConfig,
)
from neuropetlm.src.model import NeuroPetLM
from neuropetlm.src.tokenizer import CharacterTokenizer


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input/next-character pairs from a one-dimensional token stream."""
    starts = torch.randint(len(tokens) - context_length, (batch_size,))
    inputs = torch.stack([tokens[start : start + context_length] for start in starts])
    targets = torch.stack(
        [tokens[start + 1 : start + context_length + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: NeuroPetLM,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    training_config: TrainingConfig,
    model_config: ModelConfig,
    device: torch.device,
) -> dict[str, float]:
    """Estimate train and validation loss without changing model weights."""
    was_training = model.training
    model.eval()
    result: dict[str, float] = {}
    try:
        for name, source in (("train", train_tokens), ("validation", validation_tokens)):
            losses = []
            for _ in range(training_config.eval_iters):
                inputs, targets = get_batch(
                    source,
                    training_config.batch_size,
                    model_config.context_length,
                    device,
                )
                _, loss = model(inputs, targets)
                assert loss is not None
                losses.append(loss.item())
            result[name] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return result


def main() -> None:
    model_config = ModelConfig()
    training_config = TrainingConfig()
    torch.manual_seed(training_config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus_path = Path(DATA_PATH)
    corpus = corpus_path.read_text(encoding="utf-8")
    split_at = int(len(corpus) * training_config.train_fraction)
    if split_at <= model_config.context_length or len(corpus) - split_at <= model_config.context_length:
        raise ValueError("training and validation text must each exceed context_length")

    # Fit vocabulary only on the training portion to avoid validation leakage.
    tokenizer = CharacterTokenizer.fit(corpus[:split_at])
    all_tokens = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    train_tokens = all_tokens[:split_at]
    validation_tokens = all_tokens[split_at:]

    model = NeuroPetLM(tokenizer.vocab_size, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(checkpoint_dir / "tokenizer.json")

    best_validation_loss = float("inf")
    for step in range(training_config.max_iters):
        if step % training_config.eval_interval == 0 or step == training_config.max_iters - 1:
            losses = estimate_loss(
                model,
                train_tokens,
                validation_tokens,
                training_config,
                model_config,
                device,
            )
            print(
                f"step {step:04d} | train loss {losses['train']:.4f} "
                f"| validation loss {losses['validation']:.4f}"
            )
            if losses["validation"] < best_validation_loss:
                best_validation_loss = losses["validation"]
                torch.save(
                    {
                        "model_config": model_config.to_dict(),
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "validation_loss": best_validation_loss,
                    },
                    checkpoint_dir / "neuropetlm.pt",
                )

        inputs, targets = get_batch(
            train_tokens,
            training_config.batch_size,
            model_config.context_length,
            device,
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Best checkpoint saved in {checkpoint_dir / 'neuropetlm.pt'}")


if __name__ == "__main__":
    main()
