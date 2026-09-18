"""Window identities, corpus fingerprints, and local tokenizer behavior."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from src.llm.experiments.dit.data import (
    load_text_examples,
    load_tokenizer,
    make_text_examples,
)


class DataTest(unittest.TestCase):
    def test_target_inclusive_disjoint_windows_with_offsets(self):
        examples = make_text_examples(
            list(range(22)), 3, offset=1, stride=5, max_examples=3
        )
        self.assertEqual(examples.offsets, (1, 6, 11))
        self.assertEqual(
            examples.tokens.tolist(), [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]]
        )
        self.assertEqual(
            examples.example_metadata([2]),
            [{"example_id": 2, "token_offset": 11, "token_ids": [11, 12, 13, 14]}],
        )
        default = make_text_examples(range(11), 3)
        self.assertEqual(default.offsets, (0, 4))
        self.assertEqual(default.tokens.shape, (2, 4))

    def test_window_rejections(self):
        for options in (
            dict(context_length=0),
            dict(context_length=3, stride=3),
            dict(context_length=3, offset=-1),
            dict(context_length=3, max_examples=0),
            dict(context_length=50),
            dict(context_length=3, offset=100),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_text_examples(range(10), **options)
        with self.assertRaises(ValueError):
            make_text_examples([1, -2, 3], 1)
        with self.assertRaises(ValueError):
            make_text_examples([1.25, 2.75, 3], 1)

    def test_local_tokenization_fingerprints_and_no_padding_or_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tokenizer = Tokenizer(
                models.WordLevel(
                    {"<unk>": 0, "to": 1, "be": 2, "or": 3, "not": 4}, unk_token="<unk>"
                )
            )
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.enable_padding(length=20)
            tokenizer.enable_truncation(max_length=2)
            tokenizer.save(str(directory / "tokenizer.json"))
            corpus = directory / "shakespeare.txt"
            corpus.write_text("to be or not to be", encoding="utf-8")
            loaded = load_text_examples(corpus, directory, 2)
            self.assertEqual(loaded.tokens.tolist(), [[1, 2, 3], [4, 1, 2]])
            self.assertEqual(loaded.offsets, (0, 3))
            self.assertEqual(
                loaded.corpus_sha256, hashlib.sha256(corpus.read_bytes()).hexdigest()
            )
            self.assertEqual(
                loaded.tokenizer_sha256,
                hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                load_tokenizer(directory).encode("to be or not to be").ids,
                [1, 2, 3, 4, 1, 2],
            )
            corpus.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_text_examples(corpus, directory, 2)

    @unittest.skipUnless(
        os.environ.get("PLUTO_GPT2_TOKENIZER_DIR"),
        "local GPT-2 tokenizer not configured",
    )
    def test_shakespeare_with_real_gpt2_tokenizer(self):
        directory = os.environ["PLUTO_GPT2_TOKENIZER_DIR"]
        corpus = Path(__file__).resolve().parents[4] / "testdata" / "shakespeare.txt"
        loaded = load_text_examples(corpus, directory, 16, max_examples=3)
        tokenizer = load_tokenizer(directory)
        all_ids = tokenizer.encode(corpus.read_text(), add_special_tokens=False).ids
        self.assertEqual(
            loaded.tokens.tolist(), [all_ids[0:17], all_ids[17:34], all_ids[34:51]]
        )
        self.assertLess(loaded.tokens.max().item(), 50_257)
        self.assertEqual(
            tokenizer.decode(tokenizer.encode("Exeunt", add_special_tokens=False).ids),
            "Exeunt",
        )


if __name__ == "__main__":
    unittest.main()
