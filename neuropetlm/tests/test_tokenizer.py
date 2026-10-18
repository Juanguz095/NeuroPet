import json
import tempfile
import unittest
from pathlib import Path

from neuropetlm.src.tokenizer import CharacterTokenizer


class CharacterTokenizerTests(unittest.TestCase):
    def test_encode_and_decode_round_trip(self) -> None:
        text = "Hola, ¿cómo estás?\n"
        tokenizer = CharacterTokenizer.fit(text)

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_vocabulary_is_deterministic(self) -> None:
        first = CharacterTokenizer.fit("casa")
        second = CharacterTokenizer.fit("saca")

        self.assertEqual(first.vocabulary, second.vocabulary)

    def test_unknown_characters_use_unk_id(self) -> None:
        tokenizer = CharacterTokenizer.fit("gato")

        self.assertEqual(tokenizer.encode("x"), [tokenizer.unk_id])
        self.assertEqual(tokenizer.decode([tokenizer.unk_id]), "<unk>")

    def test_empty_text(self) -> None:
        tokenizer = CharacterTokenizer.fit("")

        self.assertEqual(tokenizer.vocab_size, 1)
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")

    def test_save_and_load_preserve_ids(self) -> None:
        text = "¡Hola, NeuroPet!"
        tokenizer = CharacterTokenizer.fit(text)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "tokenizer.json"
            tokenizer.save(path)
            loaded = CharacterTokenizer.load(path)

        self.assertEqual(loaded.vocabulary, tokenizer.vocabulary)
        self.assertEqual(loaded.encode(text), tokenizer.encode(text))
        self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_rejects_invalid_token_ids(self) -> None:
        tokenizer = CharacterTokenizer.fit("a")

        with self.assertRaises(ValueError):
            tokenizer.decode([tokenizer.vocab_size])
        with self.assertRaises(TypeError):
            tokenizer.decode([1.5])

    def test_load_rejects_unsupported_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tokenizer.json"
            path.write_text(
                json.dumps({"format_version": 999, "tokenizer": "character", "vocabulary": ["<unk>"]}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                CharacterTokenizer.load(path)


if __name__ == "__main__":
    unittest.main()
