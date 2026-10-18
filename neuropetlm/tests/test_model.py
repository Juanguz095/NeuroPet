import unittest

try:
    import torch
except ModuleNotFoundError:  # PyTorch is needed to execute these component tests.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class NeuroPetLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from neuropetlm.config import ModelConfig
        from neuropetlm.src.model import NeuroPetLM

        cls.ModelConfig = ModelConfig
        cls.NeuroPetLM = NeuroPetLM

    def make_model(self):
        config = self.ModelConfig(
            embedding_dim=16,
            num_heads=2,
            num_layers=2,
            context_length=8,
            dropout=0.0,
        )
        return self.NeuroPetLM(vocab_size=12, config=config)

    def test_forward_returns_logits_and_loss(self) -> None:
        model = self.make_model()
        token_ids = torch.randint(0, 12, (2, 5))
        targets = torch.randint(0, 12, (2, 5))

        logits, loss = model(token_ids, targets)

        self.assertEqual(tuple(logits.shape), (2, 5, 12))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.ndim, 0)

    def test_generation_appends_requested_number_of_tokens(self) -> None:
        model = self.make_model()
        prompt = torch.tensor([[1, 2, 3]])

        result = model.generate(prompt, max_new_tokens=4)

        self.assertEqual(tuple(result.shape), (1, 7))


if __name__ == "__main__":
    unittest.main()
