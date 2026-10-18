"""Generate text from a trained NeuroPetLM checkpoint."""

import argparse
from pathlib import Path

import torch

from neuropetlm.config import CHECKPOINT_DIR, ModelConfig
from neuropetlm.src.model import NeuroPetLM
from neuropetlm.src.tokenizer import CharacterTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with NeuroPetLM")
    parser.add_argument("--prompt", required=True, help="Starting text for generation")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_path = checkpoint_dir / "neuropetlm.pt"
    tokenizer = CharacterTokenizer.load(checkpoint_dir / "tokenizer.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Checkpoints contain only primitive metadata and a tensor state_dict.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = NeuroPetLM(tokenizer.vocab_size, model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        parser.error("--prompt must contain at least one character")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
