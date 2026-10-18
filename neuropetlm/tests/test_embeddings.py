import unittest

try:
    import torch
except ModuleNotFoundError:  # PyTorch is needed to execute these component tests.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TokenAndPositionEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from neuropetlm.src.embeddings import TokenAndPositionEmbedding

        cls.Embedding = TokenAndPositionEmbedding

    def test_output_shape(self) -> None:
        embedding = self.Embedding(vocab_size=20, embedding_dim=8, context_length=16)
        token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

        result = embedding(token_ids)

        self.assertEqual(tuple(result.shape), (2, 3, 8))

    def test_token_and_position_parameters_receive_gradients(self) -> None:
        embedding = self.Embedding(vocab_size=20, embedding_dim=8, context_length=16)
        token_ids = torch.tensor([[1, 2, 3]])

        embedding(token_ids).sum().backward()

        self.assertIsNotNone(embedding.token_embedding.weight.grad)
        self.assertIsNotNone(embedding.position_embedding.weight.grad)

    def test_rejects_sequence_longer_than_context(self) -> None:
        embedding = self.Embedding(vocab_size=20, embedding_dim=8, context_length=2)

        with self.assertRaises(ValueError):
            embedding(torch.tensor([[1, 2, 3]]))

    def test_rejects_non_batch_sequence_input(self) -> None:
        embedding = self.Embedding(vocab_size=20, embedding_dim=8, context_length=16)

        with self.assertRaises(ValueError):
            embedding(torch.tensor([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
