import unittest

try:
    import torch
except ModuleNotFoundError:  # PyTorch is needed to execute these component tests.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CausalSelfAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from neuropetlm.src.attention import CausalSelfAttention

        cls.Attention = CausalSelfAttention

    def test_output_shape(self) -> None:
        attention = self.Attention(embedding_dim=8, num_heads=2, context_length=16)
        inputs = torch.randn(3, 5, 8)

        result = attention(inputs)

        self.assertEqual(tuple(result.shape), (3, 5, 8))

    def test_future_tokens_do_not_change_earlier_outputs(self) -> None:
        attention = self.Attention(embedding_dim=8, num_heads=2, context_length=16)
        attention.eval()
        original = torch.randn(1, 4, 8)
        changed_future = original.clone()
        changed_future[:, 2:, :] += 100

        original_output = attention(original)
        changed_output = attention(changed_future)

        torch.testing.assert_close(original_output[:, :2, :], changed_output[:, :2, :])

    def test_parameters_receive_gradients(self) -> None:
        attention = self.Attention(embedding_dim=8, num_heads=2, context_length=16)
        inputs = torch.randn(1, 4, 8)

        attention(inputs).sum().backward()

        self.assertIsNotNone(attention.qkv_projection.weight.grad)
        self.assertIsNotNone(attention.output_projection.weight.grad)

    def test_rejects_incompatible_head_count(self) -> None:
        with self.assertRaises(ValueError):
            self.Attention(embedding_dim=7, num_heads=2, context_length=16)

    def test_rejects_sequence_longer_than_context(self) -> None:
        attention = self.Attention(embedding_dim=8, num_heads=2, context_length=2)

        with self.assertRaises(ValueError):
            attention(torch.randn(1, 3, 8))


if __name__ == "__main__":
    unittest.main()
